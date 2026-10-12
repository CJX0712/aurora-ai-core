/*
 * Aurora AI Core — 统一入口
 * Author: 晨星 (Morning Star)
 *
 * Node:  const Aurora = require('aurora-ai-core');
 * 浏览器: <script src="src/num.js">… 后使用全局 Aurora
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = lib;
  const g = (typeof globalThis !== 'undefined') ? globalThis : root;
  g.Aurora = Object.assign({}, g.Aurora, lib);
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const isNode = (typeof module !== 'undefined' && module.exports);
  const Num = isNode ? require('./num.js') : null;
  const ML = isNode ? require('./ml.js') : null;
  const DL = isNode ? require('./dl.js') : null;
  const Verify = isNode ? require('./verify.js') : null;
  return {
    version: '1.0.0',
    author: '晨星',
    Num: Num,
    ML: ML,
    DL: DL,
    Verify: Verify
  };
});
