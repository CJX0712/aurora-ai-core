/*
 * Aurora AI Core — Deep Learning (自动微分 + MLP + 注意力)
 * Author: 晨星 (Morning Star)
 *
 * 零重写：反向模式自动微分 (Baydin 2018, Griewank 1989) 即世界顶级
 * 梯度计算范式；注意力复用 Vaswani 2017 原始公式。全零依赖、可梯度自检。
 */
(function (root, factory) {
  const g = (typeof globalThis !== 'undefined') ? globalThis : root;
  const dep = (typeof module !== 'undefined' && module.exports) ? require('./num.js') : (g.Aurora && g.Aurora.Num);
  const lib = factory(dep);
  if (typeof module !== 'undefined' && module.exports) module.exports = lib;
  g.Aurora = g.Aurora || {};
  g.Aurora.DL = lib;
})(typeof self !== 'undefined' ? self : this, function (N) {
  'use strict';

  const zeros2d = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  const sum2d = (m) => m.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0);
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  // ---------- 反向模式自动微分 ----------
  class V {
    constructor(data, parents, backward) {
      this.data = data;            // number[][]
      this.grad = zeros2d(data.length, data[0].length);
      this._parents = parents || [];
      this._backward = backward || (() => {});
    }
    backward() {
      const topo = [];
      const visited = new Set();
      const build = (v) => {
        if (visited.has(v)) return;
        visited.add(v);
        for (const p of v._parents) build(p);
        topo.push(v);
      };
      build(this);
      this.grad = zeros2d(this.data.length, this.data[0].length);
      this.grad[0][0] = 1;
      for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
    }
  }

  // ---------- 算子 (全部 2D 张量) ----------
  function matmul(a, b) {
    const r = a.data.length, k = b.data.length, c = b.data[0].length;
    const out = new V(zeros2d(r, c), [a, b], () => {
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        const g = out.grad[i][j];
        for (let p = 0; p < k; p++) { a.grad[i][p] += g * b.data[p][j]; b.grad[p][j] += a.data[i][p] * g; }
      }
    });
    for (let i = 0; i < r; i++) { const o = out.data[i]; const ar = a.data[i]; for (let p = 0; p < k; p++) { const av = ar[p]; for (let j = 0; j < c; j++) o[j] += av * b.data[p][j]; } }
    return out;
  }
  function add(a, b) {
    const r = a.data.length, c = a.data[0].length;
    const bRow = (b.data.length === 1); // 行广播
    const out = new V(zeros2d(r, c), [a, b], () => {
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        a.grad[i][j] += out.grad[i][j];
        if (bRow) b.grad[0][j] += out.grad[i][j]; else b.grad[i][j] += out.grad[i][j];
      }
    });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[i][j] = a.data[i][j] + (bRow ? b.data[0][j] : b.data[i][j]);
    return out;
  }
  function addScalar(a, s) {
    const r = a.data.length, c = a.data[0].length;
    const out = new V(zeros2d(r, c), [a], () => { for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) a.grad[i][j] += out.grad[i][j]; });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[i][j] = a.data[i][j] + s;
    return out;
  }
  function sub(a, b) {
    const r = a.data.length, c = a.data[0].length;
    const out = new V(zeros2d(r, c), [a, b], () => { for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) { a.grad[i][j] += out.grad[i][j]; b.grad[i][j] -= out.grad[i][j]; } });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[i][j] = a.data[i][j] - b.data[i][j];
    return out;
  }
  function scale(s, a) {
    const r = a.data.length, c = a.data[0].length;
    const out = new V(zeros2d(r, c), [a], () => { for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) a.grad[i][j] += s * out.grad[i][j]; });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[i][j] = s * a.data[i][j];
    return out;
  }
  function sumAll(a) {
    const s = sum2d(a.data);
    const out = new V([[s]], [a], () => { const g = out.grad[0][0]; for (let i = 0; i < a.data.length; i++) for (let j = 0; j < a.data[0].length; j++) a.grad[i][j] += g; });
    return out;
  }
  function elemOp(a, fn, dFn, name) {
    const r = a.data.length, c = a.data[0].length;
    const out = new V(zeros2d(r, c), [a], () => { for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) a.grad[i][j] += dFn(a.data[i][j]) * out.grad[i][j]; });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[i][j] = fn(a.data[i][j]);
    return out;
  }
  const tanh = (a) => elemOp(a, Math.tanh, x => 1 - Math.tanh(x) * Math.tanh(x));
  const relu = (a) => elemOp(a, x => Math.max(0, x), x => x > 0 ? 1 : 0);
  const exp = (a) => elemOp(a, Math.exp, Math.exp);
  const log = (a) => elemOp(a, x => Math.log(x + 1e-12), x => 1 / (x + 1e-12));
  const sig = (a) => elemOp(a, sigmoid, x => { const s = sigmoid(x); return s * (1 - s); });

  // 行softmax (每行和为1)
  function softmaxRowsV(M) {
    const r = M.data.length, c = M.data[0].length;
    const out = new V(zeros2d(r, c), [M], () => {
      for (let i = 0; i < r; i++) {
        const p = out.data[i], g = out.grad[i];
        for (let a = 0; a < c; a++) {
          let d = 0;
          for (let b = 0; b < c; b++) d += (a === b ? p[a] * (1 - p[a]) : -p[a] * p[b]) * g[b];
          M.grad[i][a] += d;
        }
      }
    });
    for (let i = 0; i < r; i++) {
      const row = M.data[i];
      const m = Math.max(...row);
      const e = row.map(v => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      out.data[i] = e.map(v => v / s);
    }
    return out;
  }

  // 转置 (可微：梯度按原位置回传)
  function transpose(a) {
    const r = a.data.length, c = a.data[0].length;
    const out = new V(zeros2d(c, r), [a], () => {
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) a.grad[i][j] += out.grad[j][i];
    });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[j][i] = a.data[i][j];
    return out;
  }

  // 逐元素乘 (支持行广播)
  function mul(a, b) {
    const r = a.data.length, c = a.data[0].length;
    const bRow = (b.data.length === 1);
    const out = new V(zeros2d(r, c), [a, b], () => {
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        const bv = bRow ? b.data[0][j] : b.data[i][j];
        a.grad[i][j] += bv * out.grad[i][j];
        if (bRow) b.grad[0][j] += a.data[i][j] * out.grad[i][j];
        else b.grad[i][j] += a.data[i][j] * out.grad[i][j];
      }
    });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[i][j] = a.data[i][j] * (bRow ? b.data[0][j] : b.data[i][j]);
    return out;
  }

  // 行切片 (可微：梯度回传到源矩阵对应行)
  function sliceRows(a, s, e) {
    const r = e - s, c = a.data[0].length;
    const out = new V(zeros2d(r, c), [a], () => {
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) a.grad[s + i][j] += out.grad[i][j];
    });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[i][j] = a.data[s + i][j];
    return out;
  }

  // 通用梯度检验：返回最大相对误差 (应 ≈ 1e-7)
  function gradCheck(lossFn, params, eps) {
    eps = eps || 1e-5;
    // 清零参数梯度，避免多次调用累积
    for (const p of params) for (let i = 0; i < p.grad.length; i++) p.grad[i].fill(0);
    const loss = lossFn(params);
    loss.backward();
    const analytic = params.map(p => p.grad.flat());
    const numeric = [];
    for (const p of params) {
      const r = p.data.length, c = p.data[0].length;
      const ng = [];
      for (let idx = 0; idx < r * c; idx++) {
        const i = Math.floor(idx / c), j = idx % c;
        const old = p.data[i][j];
        p.data[i][j] = old + eps; const lp = lossFn(params).data[0][0];
        p.data[i][j] = old - eps; const lm = lossFn(params).data[0][0];
        p.data[i][j] = old;
        ng.push((lp - lm) / (2 * eps));
      }
      numeric.push(ng);
    }
    let maxRel = 0, maxAbs = 0;
    for (let k = 0; k < params.length; k++) {
      for (let i = 0; i < analytic[k].length; i++) {
        const a = analytic[k][i], n = numeric[k][i];
        const rel = Math.abs(a - n) / (Math.abs(a) + Math.abs(n) + 1e-12);
        maxRel = Math.max(maxRel, rel); maxAbs = Math.max(maxAbs, Math.abs(a - n));
      }
    }
    return { maxRel, maxAbs };
  }

  // ---------- MLP ----------
  function mlp(layerSizes, opts) {
    opts = opts || {};
    const act = opts.activation || 'tanh';
    const rnd = N.rng(opts.seed == null ? 99 : opts.seed);
    const W = [], b = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const fan = layerSizes[l] + layerSizes[l + 1];
      const w = zeros2d(layerSizes[l], layerSizes[l + 1]);
      for (let i = 0; i < layerSizes[l]; i++) for (let j = 0; j < layerSizes[l + 1]; j++) w[i][j] = (rnd() * 2 - 1) * Math.sqrt(2 / fan);
      W.push(new V(w));
      b.push(new V(zeros2d(1, layerSizes[l + 1])));
    }
    function forward(x2d) {
      let h = new V(x2d);
      for (let l = 0; l < W.length; l++) {
        h = matmul(h, W[l]);
        h = add(h, b[l]);
        if (l < W.length - 1) h = (act === 'relu') ? relu(h) : tanh(h);
      }
      return h;
    }
    function train(X, Y, opts2) {
      opts2 = opts2 || {};
      const lr = opts2.lr || 0.05, epochs = opts2.epochs || 800, batch = opts2.batch || X.length;
      const lossTrace = [];
      const n = X.length, C = layerSizes[layerSizes.length - 1];
      for (let ep = 0; ep < epochs; ep++) {
    let tot = new V([[0]]);
    for (let bi = 0; bi < n; bi += batch) {
          const xe = X.slice(bi, bi + batch), ye = Y.slice(bi, bi + batch);
      // softmax 交叉熵 —— 全在自动微分图内计算，梯度可回传
      const logits = forward(xe);
      const logSm = log(softmaxRowsV(logits));                 // N×C
      const mask = new V(ye.map(t => {                          // one-hot 选择正确类别
        const row = new Array(C).fill(0);
        row[t] = 1;
        return row;
      }));
      const L = scale(-1 / ye.length, sumAll(mul(logSm, mask)));
      tot = add(tot, L);
        }
        const loss = scale(1, tot);
        loss.backward();
        for (const p of [...W, ...b]) {
          for (let i = 0; i < p.data.length; i++) for (let j = 0; j < p.data[0].length; j++) p.data[i][j] -= lr * p.grad[i][j];
        }
        lossTrace.push(loss.data[0][0]);
      }
      return lossTrace;
    }
    function predict(x) { const o = forward([x]).data[0]; return o.indexOf(Math.max(...o)); }
    function predictBatch(X) { return X.map(x => predict(x)); }
    return { forward, train, predict, predictBatch, W, b };
  }

  // ---------- 多头注意力 (Vaswani 2017) ----------
  function multiHeadAttention(dModel, nHeads, opts) {
    opts = opts || {};
    const rnd = N.rng(opts.seed == null ? 2024 : opts.seed);
    const dk = Math.floor(dModel / nHeads);
    const projDim = nHeads * dk; // = dModel 当可整除
    const init = (r, c) => { const m = zeros2d(r, c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m[i][j] = (rnd() * 2 - 1) * Math.sqrt(2 / (r + c)); return new V(m); };
    const Wq = [], Wk = [], Wv = [], Wo = init(projDim, dModel);
    for (let h = 0; h < nHeads; h++) { Wq.push(init(dModel, dk)); Wk.push(init(dModel, dk)); Wv.push(init(dModel, dk)); }
    const params = [...Wq, ...Wk, ...Wv, Wo];
    function forward(X) { // X: N×dModel
      const outs = [];
      for (let h = 0; h < nHeads; h++) {
        const Q = matmul(new V(X), Wq[h]);           // N×dk
        const K = matmul(new V(X), Wk[h]);           // N×dk
        const Vv = matmul(new V(X), Wv[h]);          // N×dk
        const Kt = transpose(K); // dk×N (可微转置，梯度可回传至 Wk)
        const scores = matmul(Q, Kt);                // N×N
        const scaled = scale(1 / Math.sqrt(dk), scores);
        const attn = softmaxRowsV(scaled);           // 每行和=1
        const ctx = matmul(attn, Vv);                // N×dk
        outs.push(ctx);
      }
      // 逐头投影后求和 —— 与 concat+单次投影数学等价，但梯度链完整可微
      let out = new V(zeros2d(outs[0].data.length, dModel));
      for (let h = 0; h < nHeads; h++) {
        const Wo_h = sliceRows(Wo, h * dk, (h + 1) * dk); // dk×dModel
        out = add(out, matmul(outs[h], Wo_h));            // N×dModel
      }
      return out; // N×dModel
    }
    return { forward, params, Wq, Wk, Wv, Wo };
  }

  return { V, matmul, add, addScalar, sub, mul, scale, sumAll, tanh, relu, exp, log, sig, softmaxRowsV, sliceRows, transpose, gradCheck, mlp, multiHeadAttention };
});
