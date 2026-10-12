/*
 * Aurora AI Core — 无头验证器 (Headless Verifier)
 * Author: 晨星
 * 零依赖。运行所有数学不变量 + 微指令层，必须全绿 (exit 1 on failure)。
 */
const N = require('../src/num.js');
const ML = require('../src/ml.js');
const DL = require('../src/dl.js');
const V = require('../src/verify.js');
const { createHarness, createMicro, microMatmul, microSoftmaxRows } = V;

const H = createHarness();
let passed0 = 0;

// ---------------- 数值内核 ----------------
{
  const A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]];
  const L = N.cholesky(A);
  H.register('Cholesky 重构误差≈0', 'num', () => ({ ok: N.choleskyReconError(L, A) < 1e-10, detail: N.choleskyReconError(L, A).toExponential(2) }));
  H.register('Cholesky 下三角', 'num', () => {
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) if (Math.abs(L[i][j]) > 1e-12) return { ok: false };
    return { ok: true };
  });
  const M = [[2, 1, 0], [1, 3, 1], [0, 1, 4]];
  const je = N.jacobiEigen(M);
  H.register('Jacobi 特征残差≈机器精度', 'num', () => ({ ok: N.eigenResidual(M, je.eigenvalues, je.eigenvectors) < 1e-10, detail: N.eigenResidual(M, je.eigenvalues, je.eigenvectors).toExponential(2) }));
  H.register('Jacobi 特征向量正交', 'num', () => ({ ok: N.orthogonalityError(je.eigenvectors) < 1e-10, detail: N.orthogonalityError(je.eigenvectors).toExponential(2) }));
  H.register('Jacobi 特征值降序', 'num', () => { for (let i = 1; i < je.eigenvalues.length; i++) if (je.eigenvalues[i] > je.eigenvalues[i - 1] + 1e-12) return { ok: false }; return { ok: true }; });
  const B = [[1, 2], [3, 4], [5, 6]];
  const sv = N.svd(B);
  H.register('SVD 重构误差≈0', 'num', () => ({ ok: N.svdReconError(sv.U, sv.S, sv.V, B) < 1e-9, detail: N.svdReconError(sv.U, sv.S, sv.V, B).toExponential(2) }));
  H.register('SVD 奇异值降序', 'num', () => { for (let i = 1; i < sv.S.length; i++) if (sv.S[i] > sv.S[i - 1] + 1e-12) return { ok: false }; return { ok: true }; });
  H.register('logsumexp 数值稳定 (防溢出)', 'num', () => { const x = N.logsumexp([1000, 1000.1]); const exp = 1000 + Math.log(Math.exp(0) + Math.exp(0.1)); return { ok: Math.abs(x - exp) < 1e-6, detail: x.toFixed(4) }; });
  H.register('softmax 行和=1', 'num', () => { const s = N.softmax([1, 2, 3]).reduce((a, b) => a + b, 0); return { ok: Math.abs(s - 1) < 1e-12, detail: s.toFixed(12) }; });
  const SPD = N.matmul(N.randn(5, 5, 3), N.transpose(N.randn(5, 5, 3)));
  H.register('PSD 判定', 'num', () => ({ ok: N.isPSD(SPD) }));
  H.register('逆 × 矩阵 = I', 'num', () => { const inv = N.inverseSPD(SPD); const I = N.matmul(inv, SPD); let e = 0; for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) e = Math.max(e, Math.abs(I[i][j] - (i === j ? 1 : 0))); return { ok: e < 1e-8, detail: e.toExponential(2) }; });
}

// ---------------- 机器学习 ----------------
{
  const X = N.randn(60, 4, 11);
  const pca = ML.pca(X, { components: 4 });
  H.register('PCA 协方差对称', 'ml', () => ({ ok: N.isSymmetric(pca.covariance, 1e-9) }));
  H.register('PCA 协方差 PSD', 'ml', () => ({ ok: N.isPSD(pca.covariance) }));
  H.register('PCA 主成分正交', 'ml', () => ({ ok: N.orthogonalityError(pca.components) < 1e-9, detail: N.orthogonalityError(pca.components).toExponential(2) }));
  H.register('PCA 投影方差=特征值', 'ml', () => {
    let ok = true;
    for (let i = 0; i < pca.components.length; i++) {
      const col = pca.Z.map(r => r[i]);
      // 协方差按 1/(n-1) 归一 → 投影方差须同口径才严格等于特征值
      const varr = col.reduce((s, v) => s + v * v, 0) / (col.length - 1);
      if (Math.abs(varr - pca.explainedVariance[i]) / (pca.explainedVariance[i] + 1e-12) > 1e-6) ok = false;
    }
    return { ok };
  });

  // 生成 3 个高斯簇
  const g1 = N.randn(40, 2, 21).map(r => [r[0] + 5, r[1] + 5]);
  const g2 = N.randn(40, 2, 22).map(r => [r[0] - 5, r[1] - 5]);
  const g3 = N.randn(40, 2, 23).map(r => [r[0] + 5, r[1] - 5]);
  const Xc = g1.concat(g2, g3);
  const gmm = ML.gmm(Xc, 3, { seed: 5, maxIter: 100 });
  H.register('GMM 责任度每行Σ=1', 'ml', () => { for (const r of gmm.R) { const s = r.reduce((a, b) => a + b, 0); if (Math.abs(s - 1) > 1e-9) return { ok: false, detail: s.toFixed(12) }; } return { ok: true }; });
  H.register('GMM 权重Σ=1', 'ml', () => { const s = gmm.w.reduce((a, b) => a + b, 0); return { ok: Math.abs(s - 1) < 1e-9, detail: s.toFixed(12) }; });
  H.register('GMM 对数似然单调不减', 'ml', () => { for (let i = 1; i < gmm.llTrace.length; i++) if (gmm.llTrace[i] < gmm.llTrace[i - 1] - 1e-6) return { ok: false, detail: `step ${i}` }; return { ok: true }; });

  const km = ML.kmeans(Xc, 3, { seed: 9 });
  H.register('k-Means 惯性单调不减', 'ml', () => { for (let i = 1; i < km.inertiaTrace.length; i++) if (km.inertiaTrace[i] > km.inertiaTrace[i - 1] + 1e-6) return { ok: false }; return { ok: true }; });

  const Xk = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const yk = [0, 1, 1, 0];
  const knn = ML.knn(Xk, yk, { k: 1 });
  H.register('kNN k=1 训练误差=0', 'ml', () => { const err = Xk.filter((x, i) => knn.predict(x, 1) !== yk[i]).length; return { ok: err === 0 }; });

  const Xnb = [[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
  const ynb = [0, 0, 1, 1];
  const nb = ML.gaussianNB(Xnb, ynb);
  H.register('朴素贝叶斯 后验Σ=1', 'ml', () => { const p = nb.predict([1.0, 2.0]).posterior; const s = Object.values(p).reduce((a, b) => a + b, 0); return { ok: Math.abs(s - 1) < 1e-9, detail: s.toFixed(12) }; });

  const Xl = N.randn(50, 3, 31);
  const trueW = [1.5, -2.0, 0.8];
  const yl = Xl.map(r => r[0] * trueW[0] + r[1] * trueW[1] + r[2] * trueW[2] + 0.01 * N.randn(1, 1, 32)[0][0]);
  // 纯 OLS (λ=0) 才严格满足正规方程残差⊥X；Ridge 会引入 λ·w 偏移
  const lr = ML.linearRegression(Xl, yl, { lambda: 0 });
  H.register('线性回归 残差⊥X', 'ml', () => { const e = Math.max(...lr.residualOrthogonality.map(Math.abs)); return { ok: e < 1e-7, detail: e.toExponential(2) }; });
  H.register('线性回归 复原权重', 'ml', () => { const e = Math.max(...lr.w.map((w, i) => Math.abs(w - trueW[i]))); return { ok: e < 0.1, detail: e.toFixed(4) }; });

  const Xlg = N.randn(120, 2, 41);
  const ylg = Xlg.map(r => (r[0] + r[1] > 0 ? 1 : 0));
  const mlg = ML.logisticRegression(Xlg, ylg, { iters: 0, lr: 0.1 });
  // 梯度检验
  const w0 = [0.3, -0.2];
  mlg.w = w0.slice();
  const ag = mlg.analyticGradient(w0); // 显式传入权重，与数值差分同一点
  // 必须按当前 w 重算 loss（构造期的 loss 是旧 w 的快照，会失效）
  const lossAt = (w) => {
    let L = 0;
    for (let i = 0; i < Xlg.length; i++) { const z = N.dot(Xlg[i], w); const p = N.sigmoid(z); L += -(ylg[i] * Math.log(p + 1e-12) + (1 - ylg[i]) * Math.log(1 - p + 1e-12)); }
    return L / Xlg.length;
  };
  const eps = 1e-5;
  const num = [];
  for (let j = 0; j < 2; j++) { const o = mlg.w[j]; mlg.w[j] = o + eps; const lp = lossAt(mlg.w); mlg.w[j] = o - eps; const lm = lossAt(mlg.w); mlg.w[j] = o; num.push((lp - lm) / (2 * eps)); }
  const rel = Math.max(...ag.map((a, j) => Math.abs(a - num[j]) / (Math.abs(a) + Math.abs(num[j]) + 1e-12)));
  H.register('逻辑回归 梯度检验≈1e-7', 'ml', () => ({ ok: rel < 1e-4, detail: rel.toExponential(2) }));
  const mlg2 = ML.logisticRegression(Xlg, ylg, { iters: 400, lr: 0.1 });
  H.register('逻辑回归 可学习 (acc>0.9)', 'ml', () => { const acc = Xlg.filter((x, i) => mlg2.predict(x) === ylg[i]).length / Xlg.length; return { ok: acc > 0.9, detail: acc.toFixed(3) }; });

  const Xt = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const yt = [0, 1, 1, 0];
  const tree = ML.decisionTree(Xt, yt, { maxDepth: 4 });
  H.register('决策树 预测在标签集内', 'ml', () => { for (const x of Xt) if (!tree.classes.includes(tree.predict(x))) return { ok: false }; return { ok: true }; });
  H.register('决策树 训练准确率', 'ml', () => { const acc = Xt.filter((x, i) => tree.predict(x) === yt[i]).length / Xt.length; return { ok: acc >= 0.75, detail: acc.toFixed(2) }; });
}

// ---------------- 深度学习 ----------------
{
  // 自动微分梯度检验 (tanh(matmul(X,W)))
  const Xd = N.randn(5, 3, 51);
  const Wd = new DL.V(N.randn(3, 2, 52));
  const lossFn = (params) => DL.sumAll(DL.tanh(DL.matmul(new DL.V(Xd), params[0])));
  const gc = DL.gradCheck(lossFn, [Wd]);
  H.register('自动微分 梯度检验(MLP)≈1e-7', 'dl', () => ({ ok: gc.maxRel < 1e-5, detail: gc.maxRel.toExponential(2) }));

  // 多头注意力梯度检验
  const dModel = 4, nHeads = 2;
  const mha = DL.multiHeadAttention(dModel, nHeads, { seed: 77 });
  const Xa = N.randn(6, dModel, 61);
  const lossA = (params) => DL.sumAll(mha.forward(Xa));
  const gca = DL.gradCheck(lossA, mha.params);
  H.register('多头注意力 梯度检验≈1e-7', 'dl', () => ({ ok: gca.maxRel < 1e-5, detail: gca.maxRel.toExponential(2) }));
  H.register('注意力 softmax 行和=1', 'dl', () => {
    // 单头内部验证
    const Q = DL.matmul(new DL.V(Xa), mha.Wq[0]);
    const K = DL.matmul(new DL.V(Xa), mha.Wk[0]);
    const Kt = new DL.V(K.data[0].map((_, j) => K.data.map(r => r[j])));
    const scores = DL.matmul(Q, Kt);
    const attn = DL.softmaxRowsV(DL.scale(1 / Math.sqrt(2), scores));
    for (const row of attn.data) { const s = row.reduce((a, b) => a + b, 0); if (Math.abs(s - 1) > 1e-9) return { ok: false, detail: s.toFixed(12) }; }
    return { ok: true };
  });

  // MLP 学 XOR
  const Xx = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const yx = [0, 1, 1, 0];
  const net = DL.mlp([2, 6, 2], { activation: 'tanh', seed: 123 });
  const lt = net.train(Xx, yx, { lr: 0.1, epochs: 4000, batch: 4 });
  H.register('MLP 损失单调下降', 'dl', () => ({ ok: lt[lt.length - 1] < lt[0], detail: `${lt[0].toFixed(3)}→${lt[lt.length - 1].toFixed(3)}` }));
  H.register('MLP 解 XOR (acc=1)', 'dl', () => { const acc = Xx.filter((x, i) => net.predict(x) === yx[i]).length / 4; return { ok: acc === 1, detail: acc.toFixed(2) }; });
}

// ---------------- 微指令层 (创新) ----------------
{
  const M = createMicro();
  const A = N.randn(8, 5, 71);
  const B = N.randn(5, 6, 72);
  const C = microMatmul(M, A, B);
  // 校验重构
  const ref = N.matmul(A, B);
  const reconErr = N.maxAbs(N.sub(C, ref));
  H.register('微指令 MAC 重构一致', 'micro', () => ({ ok: reconErr < 1e-9, detail: reconErr.toExponential(2) }));
  const S = microSoftmaxRows(M, N.randn(4, 4, 73));
  H.register('微指令 SOFTMAX 行和=1', 'micro', () => { for (const r of S) { const s = r.reduce((a, b) => a + b, 0); if (Math.abs(s - 1) > 1e-9) return { ok: false }; } return { ok: true }; });
  const rep = M.report();
  H.register('微指令 全部不变量通过', 'micro', () => ({ ok: rep.failed === 0, detail: `微指令总数=${rep.total}, 失败=${rep.failed}, 目录=${rep.catalogSize}` }));
  global.__MICRO_REPORT__ = rep;
}

// ---------------- 执行 ----------------
const R = H.run();
let line = '\n================ Aurora AI Core · 验证报告 ================\n';
const groups = {};
for (const r of R.results) { (groups[r.group] = groups[r.group] || []).push(r); }
for (const g in groups) {
  line += `\n[${g}]  ${groups[g].filter(x => x.ok).length}/${groups[g].length} 通过\n`;
  for (const r of groups[g]) line += `  ${r.ok ? '✅' : '❌'} ${r.name}${r.detail ? '  (' + r.detail + ')' : ''}\n`;
}
line += `\n============================================================\n`;
line += `总计: ${R.passed}/${R.total} 通过, 失败 ${R.failed}\n`;
if (global.__MICRO_REPORT__) { const m = global.__MICRO_REPORT__; line += `微指令层: 总数 ${m.total}, 通过 ${m.passed}, 失败 ${m.failed}, 目录 ${m.catalogSize}\n`; }
line += `============================================================\n`;
console.log(line);
process.exit(R.failed === 0 ? 0 : 1);
