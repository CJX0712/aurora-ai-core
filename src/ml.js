/*
 * Aurora AI Core — Machine Learning
 * Author: 晨星 (Morning Star)
 *
 * 零重写：复用经典最优算法 (Pearson PCA 1901, Dempster EM 1977,
 * Lloyd k-Means 1957/1982, Cover-Hart kNN 1967, Bayes 1950,
 * CART Breiman 1984, 正规方程 Gauss)。全部带可验证数学不变量。
 */
(function (root, factory) {
  const g = (typeof globalThis !== 'undefined') ? globalThis : root;
  // Node: require；浏览器: 复用已挂载到全局的 Aurora.Num
  const dep = (typeof module !== 'undefined' && module.exports) ? require('./num.js') : (g.Aurora && g.Aurora.Num);
  const lib = factory(dep);
  if (typeof module !== 'undefined' && module.exports) module.exports = lib;
  g.Aurora = g.Aurora || {};
  g.Aurora.ML = lib;
})(typeof self !== 'undefined' ? self : this, function (N) {
  'use strict';

  // ---------- PCA (协方差对称半正定 → 特征分解) ----------
  function pca(X, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const mean = N.colMeans(X);
    const Xc = X.map(r => r.map((v, j) => v - mean[j]));
    const C = N.scale(N.matmul(N.transpose(Xc), Xc), 1 / (n - 1));
    const { eigenvalues, eigenvectors } = N.jacobiEigen(C); // 降序
    const K = opts.components || d;
    const components = eigenvectors.slice(0, K); // 列向量
    const Z = Xc.map(r => components.map(v => N.dot(r, v)));
    const totalVar = N.sum(C) / d;
    const explained = eigenvalues.slice(0, K).reduce((a, b) => a + b, 0);
    return {
      mean, components, explainedVariance: eigenvalues.slice(0, K),
      Z, covariance: C, totalVariance: totalVar,
      explainedRatio: explained / (totalVar * d)
    };
  }

  // 多元高斯对数密度 (经 Cholesky 保证数值稳定)
  function logGauss(x, mu, S) {
    const d = mu.length;
    const L = N.cholesky(S);
    let logdet = 0;
    for (let i = 0; i < d; i++) logdet += 2 * Math.log(L[i][i] || 1e-300);
    const diff = x.map((v, j) => v - mu[j]);
    const y = N.cholSolve(L, diff);
    let qf = 0;
    for (let i = 0; i < d; i++) qf += y[i] * y[i];
    return -0.5 * (d * Math.log(2 * Math.PI) + logdet + qf);
  }

  // ---------- GMM via EM ----------
  function gmm(X, K, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const maxIter = opts.maxIter || 200, tol = opts.tol || 1e-8, reg = opts.reg || 1e-6;
    const rnd = N.rng(opts.seed == null ? 7 : opts.seed);
    // k-means++ 初始化均值
    const mu = kmeansppInit(X, K, rnd);
    const Sigma = [], w = [];
    for (let k = 0; k < K; k++) { Sigma.push(N.identity(d)); w.push(1 / K); }
    let llPrev = -Infinity;
    const llTrace = [];
    let R = null;
    for (let iter = 0; iter < maxIter; iter++) {
      // E-step
      R = X.map(x => {
        const ln = mu.map((m, k) => Math.log(w[k] + 1e-300) + logGauss(x, m, Sigma[k]));
        const lse = N.logsumexp(ln);
        return ln.map(l => Math.exp(l - lse)); // 每行 Σ=1
      });
      // 对数似然
      let ll = 0;
      for (let i = 0; i < n; i++) {
        const ln = mu.map((m, k) => Math.log(w[k] + 1e-300) + logGauss(X[i], m, Sigma[k]));
        ll += N.logsumexp(ln);
      }
      llTrace.push(ll);
      if (ll - llPrev < tol && iter > 0) break;
      llPrev = ll;
      // M-step (Nk[k] = Σ_i R[i][k]，列和)
      const Nk = new Array(K).fill(0);
      for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) Nk[k] += R[i][k];
      for (let k = 0; k < K; k++) w[k] = Nk[k] / n; // Σw=1
      for (let k = 0; k < K; k++) {
        const m = new Array(d).fill(0);
        for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] += R[i][k] * X[i][j];
        for (let j = 0; j < d; j++) m[j] /= (Nk[k] || 1e-12);
        mu[k] = m;
        const S = N.zeros(d, d);
        for (let i = 0; i < n; i++) {
          const diff = X[i].map((v, j) => v - m[j]);
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] += R[i][k] * diff[a] * diff[b];
        }
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] /= (Nk[k] || 1e-12);
        for (let a = 0; a < d; a++) S[a][a] += reg; // 保证 PSD
        Sigma[k] = S;
      }
    }
    return { mu, Sigma, w, R, llTrace, K, n };
  }

  // ---------- k-Means (Lloyd) ----------
  function kmeansppInit(X, K, rnd) {
    const n = X.length, d = X[0].length;
    const centers = [X[Math.floor(rnd() * n)].slice()];
    while (centers.length < K) {
      const dist = X.map(x => Math.min(...centers.map(c => sqDist(x, c))));
      const tot = dist.reduce((a, b) => a + b, 0) || 1e-12;
      let r = rnd() * tot, idx = 0;
      for (let i = 0; i < n; i++) { r -= dist[i]; if (r <= 0) { idx = i; break; } }
      centers.push(X[idx].slice());
    }
    return centers;
  }
  function sqDist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return s; }

  function kmeans(X, K, opts) {
    opts = opts || {};
    const n = X.length;
    const maxIter = opts.maxIter || 300, seed = opts.seed == null ? 11 : opts.seed;
    const rnd = N.rng(seed);
    let centers = kmeansppInit(X, K, rnd);
    let assign = new Array(n).fill(0);
    const inertiaTrace = [];
    for (let iter = 0; iter < maxIter; iter++) {
      // assign
      let changed = false;
      for (let i = 0; i < n; i++) {
        let best = 0, bd = Infinity;
        for (let k = 0; k < K; k++) { const dd = sqDist(X[i], centers[k]); if (dd < bd) { bd = dd; best = k; } }
        if (assign[i] !== best) { assign[i] = best; changed = true; }
      }
      // inertia
      let inertia = 0;
      for (let i = 0; i < n; i++) inertia += sqDist(X[i], centers[assign[i]]);
      inertiaTrace.push(inertia);
      // update
      const sum = Array.from({ length: K }, () => new Array(X[0].length).fill(0));
      const cnt = new Array(K).fill(0);
      for (let i = 0; i < n; i++) { cnt[assign[i]]++; for (let j = 0; j < X[0].length; j++) sum[assign[i]][j] += X[i][j]; }
      for (let k = 0; k < K; k++) if (cnt[k] > 0) centers[k] = sum[k].map(s => s / cnt[k]);
      if (!changed && iter > 0) break;
    }
    let inertia = 0;
    for (let i = 0; i < n; i++) inertia += sqDist(X[i], centers[assign[i]]);
    return { centers, assign, inertia, inertiaTrace, K };
  }

  // ---------- k-NN ----------
  function knn(X, y, opts) {
    opts = opts || {};
    function predict(q, k) {
      k = k || opts.k || 5;
      const dists = X.map((x, i) => ({ i, d: sqDist(x, q) }));
      dists.sort((a, b) => a.d - b.d);
      const top = dists.slice(0, k);
      if (typeof y[0] === 'number' && !Number.isInteger(y[0])) {
        // 回归
        return top.reduce((s, t) => s + y[t.i], 0) / k;
      }
      // 分类投票 (用 Map 保留标签原始类型，避免对象键被转成字符串)
      const votes = new Map();
      for (const t of top) votes.set(y[t.i], (votes.get(y[t.i]) || 0) + 1);
      let best = null, bc = -1;
      for (const [c, n] of votes) if (n > bc) { bc = n; best = c; }
      return best;
    }
    return { predict };
  }

  // ---------- 高斯朴素贝叶斯 ----------
  function gaussianNB(X, y) {
    const classes = [...new Set(y)];
    const d = X[0].length;
    const stats = {};
    for (const c of classes) {
      const rows = X.filter((_, i) => y[i] === c);
      const prior = rows.length / X.length;
      const mean = new Array(d).fill(0), varr = new Array(d).fill(0);
      for (let j = 0; j < d; j++) { let s = 0; for (const r of rows) s += r[j]; mean[j] = s / rows.length; }
      for (let j = 0; j < d; j++) { let s = 0; for (const r of rows) s += (r[j] - mean[j]) ** 2; varr[j] = s / rows.length + 1e-9; }
      stats[c] = { prior, mean, varr };
    }
    function predict(x) {
      const logPost = {};
      for (const c of classes) {
        const st = stats[c];
        let ll = Math.log(st.prior);
        for (let j = 0; j < d; j++) ll += -0.5 * Math.log(2 * Math.PI * st.varr[j]) - 0.5 * (x[j] - st.mean[j]) ** 2 / st.varr[j];
        logPost[c] = ll;
      }
      const lse = N.logsumexp(Object.values(logPost));
      const posterior = {};
      for (const c of classes) posterior[c] = Math.exp(logPost[c] - lse); // Σ=1
      let best = classes[0];
      for (const c of classes) if (posterior[c] > posterior[best]) best = c;
      return { best, posterior };
    }
    return { predict, classes };
  }

  // ---------- 线性回归 (正规方程 + Cholesky) ----------
  function linearRegression(X, y, opts) {
    opts = opts || {};
    const lam = opts.lambda || 0;
    const n = X.length, d = X[0].length;
    const XtX = N.matmul(N.transpose(X), X);
    for (let i = 0; i < d; i++) XtX[i][i] += lam;
    const Xty = N.matmul(N.transpose(X), y.map(v => [v])).map(r => r[0]);
    const w = N.cholSolve(N.cholesky(XtX), Xty);
    const yhat = X.map(r => N.dot(r, w));
    const resid = y.map((v, i) => v - yhat[i]);
    const ortho = N.matmul(N.transpose(X), resid.map(v => [v])).map(r => r[0]); // Xᵀ resid ≈ 0
    return { w, yhat, resid, residualOrthogonality: ortho, r2: r2(y, yhat) };
  }

  // ---------- 逻辑回归 (梯度下降) ----------
  function logisticRegression(X, y, opts) {
    opts = opts || {};
    const lr = opts.lr || 0.1, iters = opts.iters || 500, lam = opts.lambda || 0;
    const n = X.length, d = X[0].length;
    let w = new Array(d).fill(0);
    for (let it = 0; it < iters; it++) {
      const g = new Array(d).fill(0);
      for (let i = 0; i < n; i++) {
        const z = N.dot(X[i], w);
        const p = N.sigmoid(z);
        const err = p - y[i];
        for (let j = 0; j < d; j++) g[j] += err * X[i][j];
      }
      for (let j = 0; j < d; j++) w[j] -= lr * (g[j] / n + lam * w[j]);
    }
    function predictProb(x) { return N.sigmoid(N.dot(x, w)); }
    function predict(x) { return predictProb(x) >= 0.5 ? 1 : 0; }
    const loss = () => {
      let L = 0;
      for (let i = 0; i < n; i++) { const p = N.sigmoid(N.dot(X[i], w)); L += -(y[i] * Math.log(p + 1e-12) + (1 - y[i]) * Math.log(1 - p + 1e-12)); }
      return L / n;
    };
    // 解析梯度 (供梯度检验；可显式传入权重，默认用训练后的权重)
    function analyticGradient(wv) {
      const ww = wv || w;
      const g = new Array(d).fill(0);
      for (let i = 0; i < n; i++) { const err = N.sigmoid(N.dot(X[i], ww)) - y[i]; for (let j = 0; j < d; j++) g[j] += err * X[i][j]; }
      return g.map(v => v / n);
    }
    return { w, predict, predictProb, loss: loss(), analyticGradient };
  }

  // ---------- CART 决策树 ----------
  function gini(counts, total) {
    let s = 0;
    for (const c in counts) s += (counts[c] / total) ** 2;
    return 1 - s;
  }
  // 多数投票 (用 Map 保留标签原始类型，避免对象键被转成字符串)
  function majority(labels) {
    const m = new Map();
    for (const l of labels) m.set(l, (m.get(l) || 0) + 1);
    let best = labels[0], bc = -1;
    for (const [c, n] of m) if (n > bc) { bc = n; best = c; }
    return best;
  }
  function decisionTree(X, y, opts) {
    opts = opts || {};
    const maxDepth = opts.maxDepth || 8, minSplit = opts.minSplit || 2;
    const classes = [...new Set(y)];
    function build(rows, depth, defaultPred) {
      const idx = rows;
      if (idx.length === 0) return { leaf: true, pred: defaultPred, n: 0 };
      const labels = idx.map(i => y[i]);
      const pure = labels.every(l => l === labels[0]);
      if (pure || depth >= maxDepth || idx.length < minSplit) {
        return { leaf: true, pred: majority(labels), n: idx.length };
      }
      const d = X[0].length;
      let bestGain = -1, bestFeat = -1, bestThr = 0;
      const baseGini = (() => { const c = {}; for (const l of labels) c[l] = (c[l] || 0) + 1; return gini(c, idx.length); })();
      for (let f = 0; f < d; f++) {
        const vals = [...new Set(idx.map(i => X[i][f]))].sort((a, b) => a - b);
        for (let t = 0; t < vals.length - 1; t++) {
          const thr = (vals[t] + vals[t + 1]) / 2;
          const lc = {}, rc = {}; let ln = 0, rn = 0;
          for (const i of idx) { if (X[i][f] <= thr) { lc[y[i]] = (lc[y[i]] || 0) + 1; ln++; } else { rc[y[i]] = (rc[y[i]] || 0) + 1; rn++; } }
          if (ln === 0 || rn === 0) continue;
          const g = baseGini - (ln / idx.length) * gini(lc, ln) - (rn / idx.length) * gini(rc, rn);
          if (g > bestGain) { bestGain = g; bestFeat = f; bestThr = thr; }
        }
      }
      if (bestFeat < 0) return { leaf: true, pred: majority(labels), n: idx.length };
      const left = [], right = [];
      for (const i of idx) { if (X[i][bestFeat] <= bestThr) left.push(i); else right.push(i); }
      // 本节点多数类，作为空子节点的兜底预测
      const nodePred = majority(labels);
      return { leaf: false, feat: bestFeat, thr: bestThr, left: build(left, depth + 1, nodePred), right: build(right, depth + 1, nodePred) };
    }
    const root = build([...Array(X.length).keys()], 0, classes[0]);
    function predict(x) {
      let node = root;
      while (!node.leaf) node = (x[node.feat] <= node.thr) ? node.left : node.right;
      return node.pred;
    }
    return { predict, root, classes };
  }

  // ---------- utils ----------
  function r2(y, yhat) {
    const my = y.reduce((a, b) => a + b, 0) / y.length;
    let ssRes = 0, ssTot = 0;
    for (let i = 0; i < y.length; i++) { ssRes += (y[i] - yhat[i]) ** 2; ssTot += (y[i] - my) ** 2; }
    return 1 - ssRes / (ssTot || 1e-12);
  }

  return { pca, gmm, kmeans, knn, gaussianNB, linearRegression, logisticRegression, decisionTree, logGauss };
});
